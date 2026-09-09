"""Sharding-only schedule: one native fixture, three timing trials, one profile."""
from pathlib import Path
import json


def campaign(out, manifest, launch, args):
    settings = manifest['settings']
    def fixture(n, p, smoke=False):
        label = '{}-n{}-p{}'.format('smoke' if smoke else 'fixture',n,p)
        folder = out/'fixtures'/label
        result = out/'raw'/(label+'.json')
        options = ['--fixture-only','--subdivisions',n,'--trial',0,'--output',folder,'--result',result,
                   '--warmup',1 if smoke else 10,'--iterations',100,'--repeats',2 if smoke else 10]
        ok = launch('dolfinx',p,label,options,result,memory=True)
        return folder/('n{}'.format(n))/('{}ranks'.format(p)) if ok else None
    def replay(n,p,trial,folder,profile=False,smoke=False):
        name = '{}-n{}-p{}-trial{}'.format('smoke-sharding' if smoke else 'profile' if profile else 'timing',n,p,trial)
        result = out/'raw'/(name+'.json')
        return launch('sharding',p,name,[folder,'--backend','sharding','--trial',trial,'--result',result],result,
                      memory=profile or smoke)
    smoke_ok = True
    for p in settings['ranks']:
        folder = fixture(4,p,True)
        smoke_ok = (replay(4,p,0,folder,smoke=True) if folder else False) and smoke_ok
    if not args.skip_regression:
        launch('regression',4,'regression',[],timeout=900)
    planned = []
    for n in settings['sizes']:
        for p in settings['ranks']:
            planned.append('fixture-n{}-p{}'.format(n,p))
            planned.extend('timing-n{}-p{}-trial{}'.format(n,p,t) for t in range(3))
            planned.append('profile-n{}-p{}-trial0'.format(n,p))
    (out/'planned.json').write_text(json.dumps(planned,indent=2)+'\n')
    if smoke_ok and not args.smoke_only:
        for n in settings['sizes']:
            fixtures = {p:fixture(n,p) for p in settings['ranks']}
            for trial in range(3):
                for p in settings['ranks'][::1 if trial%2==0 else -1]:
                    if fixtures[p]: replay(n,p,trial,fixtures[p])
            for p in settings['ranks']:
                if fixtures[p]: replay(n,p,0,fixtures[p],profile=True)
    launches = json.loads((out/'launches.json').read_text())
    seen = {r['name'] for r in launches}
    for name in planned:
        if name not in seen:
            launches.append(dict(name=name,exit_code=None,skipped_reason=
                'Smoke-only requested' if args.smoke_only else 'Smoke validation failed' if not smoke_ok else 'Fixture unavailable'))
    (out/'launches.json').write_text(json.dumps(launches,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(smoke_passed=smoke_ok,
        planned=len(planned),failures=[r['name'] for r in launches if r.get('exit_code') != 0]),indent=2)+'\n')

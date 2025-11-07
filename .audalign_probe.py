import inspect, sys
try:
    import audalign
except Exception as e:
    print('IMPORT-ERROR', e)
    sys.exit(0)
print('MOD', audalign.__name__, getattr(audalign,'__version__','?'))
attrs = dir(audalign)
print('ATTRS', len(attrs))
for name in attrs:
    if 'align' in name.lower() or 'Align' in name:
        obj = getattr(audalign, name)
        print('SYM', name, type(obj))
        doc = inspect.getdoc(obj)
        if doc:
            print('DOC', name, doc.splitlines()[0][:200])
        try:
            src = inspect.getsource(obj)
            print('SRCHEAD', name, src.splitlines()[0])
        except Exception as e:
            pass
# Look for classes likely exposing methods
for name in attrs:
    obj = getattr(audalign, name)
    if inspect.isclass(obj):
        methods = [m for m in dir(obj) if 'align' in m.lower() or 'fit' in m.lower() or 'process' in m.lower()]
        if methods:
            print('CLASS', name, 'METHODS', methods[:10])

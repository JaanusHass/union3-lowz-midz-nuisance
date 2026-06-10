import sys, yaml
src, dst, out = sys.argv[1], sys.argv[2], sys.argv[3]
covmat = sys.argv[4] if len(sys.argv) > 4 else None
with open(src) as f: info = yaml.safe_load(f)
info['params']['w']  = -1      # fikseeri w0 = -1
info['params']['wa'] =  0      # fikseeri wa = 0  -> LCDM
info['output'] = out
if covmat:
    info.setdefault('sampler', {}).setdefault('mcmc', {})['covmat'] = covmat
with open(dst, 'w') as f: yaml.safe_dump(info, f, default_flow_style=False, sort_keys=False)
print('wrote', dst, '| output:', out, '| covmat:', covmat)

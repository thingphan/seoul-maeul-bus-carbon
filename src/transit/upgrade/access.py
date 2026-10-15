"""Accessibility summaries retaining newly disconnected cells."""
import numpy as np

def summarize_access(base,after,elderly,prefix,affected_min=1.,long_walk_min=10.):
    base=np.asarray(base);after=np.asarray(after);elderly=np.asarray(elderly)
    finite=np.isfinite(base)&np.isfinite(after)
    lost=np.isfinite(base)&~np.isfinite(after)
    with np.errstate(invalid='ignore'):
        added=np.where(finite,after-base,0.)
    hit=finite & (added>=affected_min)
    affected=float(elderly[hit|lost].sum())
    finite_affected=float(elderly[hit].sum())
    return {f'{prefix}_elderly_affected':affected,
            f'{prefix}_added_min_mean':float(np.average(added[hit],weights=elderly[hit])) if finite_affected>0 else 0.,
            f'{prefix}_elderly_over_10min_new':float(elderly[np.isfinite(base)&(base<=long_walk_min)&(after>long_walk_min)].sum()),
            f'{prefix}_elderly_disconnected_new':float(elderly[lost].sum()),
            f'{prefix}_elderly_unreachable_baseline':float(elderly[~np.isfinite(base)].sum())}

import threading


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return

def do_countuniq(df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    if show_agg:
        print("Counting unqiue ", counted, " by ", group_cols, '...')
    # print('the Id of train_df while function before merge: ',id(df)) # the same with train_df
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    # print('the Id of train_df while function after merge: ',id(df)) # id changes
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    return (df)
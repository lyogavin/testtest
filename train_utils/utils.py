import threading
import logging
from contextlib import contextmanager
import time, os, gc, psutil
#import lightgbm as lgb

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('./train_and_predict.log')
formatter = logging.Formatter(
    '%(asctime)s P%(process)d T%(thread)d %(levelname)s [%(module)s]:%(lineno)d %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

#lgb.logger.addHandler(hdlr)


def getLogger():
    return logger


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
        logger.debug("Counting unqiue ", counted, " by ", group_cols, '...')
    # logger.debug('the Id of train_df while function before merge: ',id(df)) # the same with train_df
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    #logger.debug(df.tail().to_string())

    df = df.merge(gp, on=group_cols, how='left', copy=False)
    # logger.debug('the Id of train_df while function after merge: ',id(df)) # id changes
    #logger.debug(df.tail().to_string())

    del gp
    if show_max:
        logger.debug(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    return (df)


def do_cumcount(df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    if show_agg:
        logger.debug("Cumulative count by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    if show_max:
        logger.debug(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    return (df)


def do_mean(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    if show_agg:
        logger.debug("Calculating mean of ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        logger.debug(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    return (df)


def do_var(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    if show_agg:
        logger.debug("Calculating variance of ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    logger.debug('dump inter var {}: {}'.format(agg_name, gp.query('A0 != 0').tail(500).to_string()))

    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        logger.debug(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type, copy=False)
    return (df)



def cpuStats(pp = False):
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    gc.collect()
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.log.debug_(sum1)(Pdb) import objgraph

    return memoryUse

@contextmanager
def timer(name, level=logging.DEBUG):
    t0 = time.time()
    logger.log(level, 'starting: [%s]',name)
    yield
    logger.log(level, '[{}] done in {} s, ended mem:{}'.format(name, time.time() - t0, cpuStats()))



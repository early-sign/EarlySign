from earlysign.schema.ES3.Base import Protocol as BaseProtocol
from earlysign.schema.ES3.GST.Method import MethodSpec
from earlysign.schema.ES3.GST.Task import TaskSpec


class Protocol(BaseProtocol):
    task: TaskSpec
    method: MethodSpec

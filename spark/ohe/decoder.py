from pyspark import keyword_only
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark.ml.wrapper import JavaTransformer


class OneHotDecoder(JavaTransformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    OneHotDecoder Custom Scala / Python Wrapper
    """
    _classpath = 'com.clarifyhealth.ohe.decoder.OneHotDecoder'

    oheSuffix = Param(Params._dummy(), 'oheSuffix', 'oheSuffix', typeConverter=TypeConverters.toString)
    idxSuffix = Param(Params._dummy(), 'idxSuffix', 'idxSuffix', typeConverter=TypeConverters.toString)
    unknownSuffix = Param(Params._dummy(), 'unknownSuffix', 'unknownSuffix', typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, oheSuffix=None, idxSuffix=None, unknownSuffix=None):
        super(OneHotDecoder, self).__init__()
        self._java_obj = self._new_java_obj(OneHotDecoder._classpath, self.uid)

        self._setDefault(oheSuffix="_OHE", idxSuffix="_IDX", unknownSuffix="Unknown")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    # noinspection PyPep8Naming
    @keyword_only
    def setParams(self, oheSuffix=None, idxSuffix=None, unknownSuffix=None):
        """
        Set the params for the OneHotDecoder
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def setOheSuffix(self, value):
        self._set(oheSuffix=value)
        return self

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def getOheSuffix(self):
        return self.getOrDefault(self.oheSuffix)

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def setIdxSuffix(self, value):
        self._set(idxSuffix=value)
        return self

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def getIdxSuffix(self):
        return self.getOrDefault(self.idxSuffix)

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def setUnknownSuffix(self, value):
        self._set(unknownSuffix=value)
        return self

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def getUnknownSuffix(self):
        return self.getOrDefault(self.unknownSuffix)

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    def getName(self):
        return "one_hot_decoder"

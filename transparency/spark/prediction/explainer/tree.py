from pyspark import keyword_only
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark.ml.wrapper import JavaTransformer


class EnsembleTreeExplainTransformer(JavaTransformer, DefaultParamsReadable,
                                     DefaultParamsWritable):
    """
    EnsembleTreeExplainTransformer Custom Scala / Python Wrapper
    """

    _classpath = 'com.clarifyhealth.prediction.explainer.EnsembleTreeExplainTransformer'

    predictionView = Param(Params._dummy(), "predictionView", "predictionView",
                           typeConverter=TypeConverters.toString)

    featureImportanceView = Param(Params._dummy(), "featureImportanceView", "featureImportanceView",
                                  typeConverter=TypeConverters.toString)

    modelPath = Param(Params._dummy(), "modelPath", "modelPath",
                      typeConverter=TypeConverters.toString)

    label = Param(Params._dummy(), "label", "label",
                  typeConverter=TypeConverters.toString)

    dropPathColumn = Param(Params._dummy(), "dropPathColumn",
                           "dropPathColumn", typeConverter=TypeConverters.toBoolean)

    isClassification = Param(Params._dummy(), "isClassification",
                             "isClassification", typeConverter=TypeConverters.toBoolean)

    ensembleType = Param(Params._dummy(), "ensembleType",
                         "ensembleType", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, predictionView=None, featureImportanceView=None, modelPath=None, label=None, dropPathColumn=None,
                 isClassification=None, ensembleType=None):
        super(EnsembleTreeExplainTransformer, self).__init__()
        self._java_obj = self._new_java_obj(EnsembleTreeExplainTransformer._classpath, self.uid)

        self._setDefault(predictionView="prediction", featureImportanceView="featureImportance", modelPath="modelPath",
                         label="test", dropPathColumn=True, isClassification=False, ensembleType="rf")

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    # noinspection PyPep8Naming
    @keyword_only
    def setParams(self, predictionView=None, featureImportanceView=None, modelPath=None, label=None,
                  dropPathColumn=None,
                  isClassification=None, ensembleType=None):
        """
        Set the params for the EnsembleTreeExplainTransformer
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # noinspection PyPep8Naming
    def setPredictionView(self, value):
        return self._set(predictionView=value)

    # noinspection PyPep8Naming
    def getPredictionView(self):
        return self.getOrDefault(self.predictionView)

    # noinspection PyPep8Naming
    def setFeatureImportanceView(self, value):
        return self._set(featureImportanceView=value)

    # noinspection PyPep8Naming
    def getFeatureImportanceView(self):
        return self.getOrDefault(self.featureImportanceView)

    # noinspection PyPep8Naming
    def setModelPath(self, value):
        return self._set(modelPath=value)

    # noinspection PyPep8Naming
    def getModelPath(self):
        return self.getOrDefault(self.modelPath)

    # noinspection PyPep8Naming
    def setLabel(self, value):
        return self._set(label=value)

    # noinspection PyPep8Naming
    def getLabel(self):
        return self.getOrDefault(self.label)

    # noinspection PyPep8Naming
    def setDropPathColumn(self, value):
        return self._set(dropPathColumn=value)

    # noinspection PyPep8Naming
    def getDropPathColumn(self):
        return self.getOrDefault(self.dropPathColumn)

    # noinspection PyPep8Naming
    def setIsClassification(self, value):
        return self._set(isClassification=value)

    # noinspection PyPep8Naming
    def getIsClassification(self):
        return self.getOrDefault(self.isClassification)

    # noinspection PyPep8Naming
    def setEnsembleType(self, value):
        return self._set(ensembleType=value)

    # noinspection PyPep8Naming
    def getEnsembleType(self):
        return self.getOrDefault(self.ensembleType)

    # noinspection PyMethodMayBeStatic,PyMissingOrEmptyDocstring,PyPep8Naming
    def getName(self) -> str:
        return "tree_explain_" + self.getPredictionView() + "_" + self.getLabel()

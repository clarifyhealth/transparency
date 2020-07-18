from pyspark import keyword_only
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark.ml.wrapper import JavaTransformer


class GLMExplainTransformer(JavaTransformer, DefaultParamsReadable,
                            DefaultParamsWritable):
    """
    GLMExplainTransformer Custom Scala / Python Wrapper
    """

    _classpath = 'com.clarifyhealth.prediction.explainer.GLMExplainTransformer'

    predictionView = Param(Params._dummy(), "predictionView", "predictionView",
                           typeConverter=TypeConverters.toString)

    coefficientView = Param(Params._dummy(), "coefficientView", "coefficientView",
                            typeConverter=TypeConverters.toString)
    linkFunctionType = Param(Params._dummy(), "linkFunctionType",
                             "linkFunctionType", typeConverter=TypeConverters.toString)
    nested = Param(Params._dummy(), "nested",
                   "nested", typeConverter=TypeConverters.toBoolean)
    calculateSum = Param(Params._dummy(), "calculateSum",
                         "calculateSum", typeConverter=TypeConverters.toBoolean)
    label = Param(Params._dummy(), "label", "label",
                  typeConverter=TypeConverters.toString)

    family = Param(Params._dummy(), "family", "family",
                   typeConverter=TypeConverters.toString)
    variancePower = Param(Params._dummy(), "variancePower", "The power in the variance function " +
                          "of the Tweedie distribution which characterizes the relationship " +
                          "between the variance and mean of the distribution. Only applicable " +
                          "for the Tweedie family. Supported values: 0 and [1, Inf).",
                          typeConverter=TypeConverters.toFloat)
    linkPower = Param(Params._dummy(), "linkPower", "The index in the power link function. " +
                      "Only applicable to the Tweedie family.",
                      typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self, predictionView=None, coefficientView=None, linkFunctionType=None, label=None, nested=None,
                 calculateSum=None, family=None, variancePower=None, linkPower=None):
        super(GLMExplainTransformer, self).__init__()
        self._java_obj = self._new_java_obj(GLMExplainTransformer._classpath, self.uid)

        self._setDefault(predictionView="prediction", coefficientView="coefficient", linkFunctionType="powerHalfLink",
                         label="test", nested=False, calculateSum=False, family="gaussian",
                         variancePower=-1.0, linkPower=0.0)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    # noinspection PyPep8Naming
    @keyword_only
    def setParams(self, predictionView=None, coefficientView=None, linkFunctionType=None, label=None, nested=None,
                  calculateSum=None, family=None, variancePower=None, linkPower=None):
        """
        Set the params for the GLMExplainTransformer
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
    def setCoefficientView(self, value):
        return self._set(coefficientView=value)

    # noinspection PyPep8Naming
    def getCoefficientView(self):
        return self.getOrDefault(self.coefficientView)

    # noinspection PyPep8Naming
    def setLinkFunctionType(self, value):
        return self._set(linkFunctionType=value)

    # noinspection PyPep8Naming
    def getLinkFunctionType(self):
        return self.getOrDefault(self.linkFunctionType)

    # noinspection PyPep8Naming
    def setNested(self, value):
        return self._set(nested=value)

    # noinspection PyPep8Naming
    def getNested(self):
        return self.getOrDefault(self.nested)

    # noinspection PyPep8Naming
    def setCalculateSum(self, value):
        return self._set(calculateSum=value)

    # noinspection PyPep8Naming
    def getCalculateSum(self):
        return self.getOrDefault(self.calculateSum)

    # noinspection PyPep8Naming
    def setLabel(self, value):
        return self._set(label=value)

    # noinspection PyPep8Naming
    def getLabel(self):
        return self.getOrDefault(self.label)

    # noinspection PyPep8Naming
    def setFamily(self, value):
        return self._set(family=value)

    # noinspection PyPep8Naming
    def getFamily(self):
        return self.getOrDefault(self.family)

    # noinspection PyPep8Naming
    def setVariancePower(self, value):
        """
        Sets the value of :py:attr:`variancePower`.
        """
        return self._set(variancePower=value)

    def getVariancePower(self):
        """
        Gets the value of variancePower or its default value.
        """
        return self.getOrDefault(self.variancePower)

    # noinspection PyPep8Naming
    def setLinkPower(self, value):
        """
        Sets the value of :py:attr:`linkPower`.
        """
        return self._set(linkPower=value)

    # noinspection PyPep8Naming
    def getLinkPower(self):
        """
        Gets the value of linkPower or its default value.
        """
        return self.getOrDefault(self.linkPower)

    # noinspection PyMethodMayBeStatic,PyMissingOrEmptyDocstring,PyPep8Naming
    def getName(self) -> str:
        return "glm_explain_" + self.getPredictionView() + "_" + self.getLabel()

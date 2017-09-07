package ai

fun logisticSigmoid(x: Double): Double {
	return 1.0 / (1.0 + Math.exp(-x))
}

fun logisticSigmoidDerivative(x: Double): Double {
	return logisticSigmoid(x) * (1 - logisticSigmoid(x))
}
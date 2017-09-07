package ai

import math.linearalgebra.Matrixd
import math.linearalgebra.Vectord
import util.RANDOM
import util.astGreater
import util.range
import java.util.*



class Ffnn(val nInputs: Int, private val nHiddenLayers: Int, private val hiddenLayerSize: Int,
           val nOutputs: Int = 1) {
	val weightMatrices = ArrayList<Matrixd>()
	init {
		astGreater(nHiddenLayers, 0)
		weightMatrices.add(Matrixd(hiddenLayerSize, nInputs))
		for (_i in 1..nHiddenLayers) {
			weightMatrices.add(Matrixd(hiddenLayerSize, hiddenLayerSize))
		}
		weightMatrices.add(Matrixd(nOutputs, hiddenLayerSize))
	}
	
	fun initializeWeights(randomFunction: () -> Double = { RANDOM.nextDouble()}) {
		for (matrix in weightMatrices) {
			for (x in range(matrix.width)) {
				for (y in range(matrix.height)) {
					matrix[y,x] = randomFunction()
				}
			}
		}
	}
	
	fun applyToTestInput(testInput: Vectord): Vectord {
		var currentVector = testInput
		for (weights in weightMatrices) {
			currentVector = weights * currentVector
			currentVector.map(::logisticSigmoid)
		}
		return currentVector
	}
	
	fun train(trainingVectors: ArrayList<Vectord>, targetOutputs: ArrayList<Vectord>) {
		
	}
}
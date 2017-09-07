package math.linearalgebra

import extensions.sequences.seq
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.RealVector
import util.astTrue
import util.range

class Vectord(val size: Int, fillFunction: (Int) -> Double = { 0.0 }) {
	
	constructor(c: Collection<Double>) : this(c.size) {
		val iter = c.iterator()
		for (i in 0..size - 1) {
			this[i] = iter.next()
		}
	}
	
	constructor(v: RealVector) : this(v.dimension) {
		for (x in 0..size - 1) {
			this[x] = v.getEntry(x)
		}
	}
	
	val data = Array(size, fillFunction)
	
	operator fun get(index: Int) = data[index]
	operator fun set(index: Int, value: Double) {
		data[index] = value
	}
	
	operator fun unaryMinus() = Vectord(size, { -this[it] })
	operator fun plus(you: Vectord) = Vectord(size, { this[it] + you[it] })
	operator fun minus(you: Vectord) = Vectord(size, { this[it] - you[it] })
	operator fun times(you: Double) = Vectord(size, { this[it] * you })
	operator fun mod(you: Double) = Vectord(size, { this[it] % you })
	operator fun div(you: Double) = Vectord(size, { this[it] / you })
	operator fun times(you: Matrixd): Vectord {
		astTrue(this.size == you.height)
		return Vectord(you.width, { x ->
			0.seq(you.height - 1).map { y ->
				this[y] * you[x, y]
			}.sum()
		})
	}
	
	override fun toString(): String {
		var text = ""
		for (x in 0..size - 1) {
			text += this[x].toString() + " "
		}
		return text.removeSuffix(" ")
	}
	
	private fun asSequence(): Sequence<Double> {
		return data.asSequence()
	}
	
	fun map(function: (Double) -> Double) {
		for (i in range(size)) {
			this[i] = function(this[i])
		}
	}
	
	fun sum(): Double = data.asSequence().sum()
	fun toArrayRealVector(): ArrayRealVector {
		val res = ArrayRealVector(size)
		for (x in 0..size - 1)
			res.setEntry(x, this[x])
		return res
	}
}

operator fun Double.times(you: Vectord) = Vectord(you.size, { you[it] * this })
package math.linearalgebra

import extensions.even
import extensions.sequences.seq
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.LUDecomposition
import util.astEqual
import util.astGreaterEqual
import util.astTrue
import util.range

// indexed (y,x)
class Matrixd(val height: Int, val width: Int = height, initFunction: (Int, Int) -> Double = { y, x -> 0.0 }) {
	
	val data = Array(height, { y -> Array(width, { x -> initFunction(y, x) }) })
	
	constructor(you: Matrixd) : this(you.height, you.width, { y, x -> you[y, x] })
	
	operator fun get(y: Int, x: Int) = data[y][x]
	
	operator fun set(y: Int, x: Int, value: Double) {
		data[y][x] = value
	}
	
	operator fun unaryMinus() = Matrixd(height, width, { y, x -> -this[y, x] })
	operator fun plus(you: Matrixd) = Matrixd(height, width, { y, x -> this[y, x] + you[y, x] })
	operator fun minus(you: Matrixd) = Matrixd(height, width, { y, x -> this[y, x] - you[y, x] })
	operator fun times(you: Double) = Matrixd(height, width, { y, x -> this[y, x] * you })
	operator fun mod(you: Double) = Matrixd(height, width, { y, x -> this[y, x] % you })
	operator fun div(you: Double) = this * (1 / you)
	operator fun times(you: Matrixd): Matrixd {
		astTrue(this.width == you.height)
		return Matrixd(this.height, you.width, { y, x ->
			0.seq(width - 1).map {
				this[y, it] * you[it, x]
			}.sum()
		})
	}
	
	operator fun times(you: Vectord): Vectord {
		astTrue(this.width == you.size)
		return Vectord(this.height, { y ->
			0.seq(width - 1).map { x ->
				this[y, x] * you[x]
			}.sum()
		})
	}
	
	fun modPow(exponent: Long, m: Double = Double.MAX_VALUE): Matrixd {
		astGreaterEqual(exponent, 0L)
		astEqual(height, width)
		var x = Matrixd.identity(height)
		if (exponent == 0L) {
			return x
		}
		var y = this
		var n = exponent
		while (n > 1) {
			if (n.even()) {
				y *= y
				n /= 2
			} else {
				x = y * x
				x %= m
				y *= y
				n = (n - 1) / 2
			}
			y %= m
		}
		return (y * x) % m
	}
	
	override fun toString(): String {
		var text = ""
		for (y in 0..height - 1) {
			var row = ""
			for (x in 0..width - 1) {
				row += this[y, x].toString() + " "
			}
			text += row.removeSuffix(" ") + "\n"
		}
		return text.removeSuffix("\n")
	}
	
	fun asSequence(): Sequence<Triple<Int, Int, Double>> {
		return 0.seq(height - 1).map { y ->
			0.seq(width - 1).map { x ->
				Triple(y, x, this[y, x])
			}
		}.flatten()
	}
	
	companion object {
		fun identity(height: Int): Matrixd {
			return Matrixd(height, height, { y, x ->
				if (y == x)
					return@Matrixd 1.0
				0.0
			})
		}
		
		fun diagonal(diagonal: Vectord): Matrixd {
			return Matrixd(diagonal.size, diagonal.size, { y, x ->
				if (y == x) diagonal[y] else 0.0
			})
		}
	}
	
	fun sum(): Double {
		var sum = 0.0
		for (y in range(height)) {
			for (x in range(width)) {
				sum += this[y, x]
			}
		}
		return sum
	}
	
	inline fun update(transformation: (Int, Int, Double) -> Double) {
		for (y in 0..height - 1) {
			for (x in 0..width - 1) {
				this[y, x] = transformation(y, x, this[y, x])
			}
		}
	}
	
	fun solveLinearEquation(rhs: Vectord): Vectord {
		astTrue(height == width)
		astTrue(width == rhs.size)
		val matrix = toArray2DRowRealMatrix()
		val constants = rhs.toArrayRealVector()
		val solver = LUDecomposition(matrix).solver
		val solution = solver.solve(constants)
		return Vectord(solution)
	}
	
	fun toArray2DRowRealMatrix(): Array2DRowRealMatrix {
		val matrix = Array2DRowRealMatrix(height, height)
		for (y in 0..height - 1) {
			for (x in 0..height - 1) {
				matrix.setEntry(y, x, this[y, x])
			}
		}
		return matrix
	}
	
	
}

operator fun Double.times(you: Matrixd) = Matrixd(you.height, you.width, { y, x -> you[y, x] * this })

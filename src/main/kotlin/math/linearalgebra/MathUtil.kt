package math.linearalgebra

object MathUtil {
	
	val EPSILON = 0.000001
	
	fun toDegree(radian: Double): Double {
		return radian / (Math.PI * 2) * 360.0
	}
	
	fun toRadian(degree: Double): Double {
		return degree * (Math.PI * 2) / 360.0
	}
	
	fun interpolate(a: Double, b: Double, t: Double): Double {
		return (1 - t) * a + t * b
	}
	
	fun clamp(s: Int, min: Int, max: Int): Int {
		if (s < min) return min
		if (s > max) return max
		return s
	}
	
	fun clamp(s: Double, min: Double, max: Double): Double {
		if (s < min) return min
		if (s > max) return max
		return s
	}
	
	fun randomInt(min: Int, max: Int): Int {
		return (Math.random() * (max - min + 1)).toInt() + min
	}
	
	fun randomDouble(min: Double, max: Double): Double {
		return Math.random() * (max - min) + min
	}
	
	fun almostEqual(a: Double, b: Double): Boolean {
		return Math.abs(b - a) < EPSILON
		
	}
	
	fun clamp(d: Double): Double {
		return clamp(d, 0.0, 1.0)
	}
	
	fun unsignByte(b: Byte): Int {
		return b.toInt() and 0xFF
	}
	
	fun byteToDouble(b: Byte): Double {
		val i = unsignByte(b)
		return i / 255.0
	}
	
	
}

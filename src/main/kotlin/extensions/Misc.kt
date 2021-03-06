package extensions

infix fun Int.divides(you: Int): Boolean {
	if (this == 0)
		return you == 0
	return you % this == 0
}

infix fun Int.divides(you: Long): Boolean {
	if (this == 0)
		return you == 0L
	return you % this == 0L
}

infix fun Long.divides(you: Int): Boolean {
	if (this == 0L)
		return you == 0
	return you % this == 0L
}

infix fun Long.divides(you: Long): Boolean {
	if (this == 0L)
		return you == 0L
	return you % this == 0L
}

fun Long.posmod(you: Int): Long {
	val tmp = this % you
	if (tmp < 0)
		return tmp + you
	return tmp
}

fun Int.posmod(you: Int): Int {
	val tmp = this % you
	if (tmp < 0)
		return tmp + you
	return tmp
}

fun Long.posmod(you: Long): Long {
	val tmp = this % you
	if (tmp < 0)
		return tmp + you
	return tmp
}

infix fun Boolean.and(you: Boolean): Boolean {
	return this && you
}

infix fun Boolean.or(you: Boolean): Boolean {
	return this || you
}


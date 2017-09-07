package util

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.EOFException
import java.io.FileInputStream
import java.nio.file.Files
import java.nio.file.Path

fun Sequence<String>.writeLinesToFile(path: Path) {
	val writer = Files.newBufferedWriter(path)
	for (line in this) {
		writer.write("$line\n")
	}
	writer.close()
}

// reads lines and maps them until map returns null
fun <T : Any> seqLinesOfFile(path: Path, map: (String) -> T?): Sequence<T> {
	val reader = Files.newBufferedReader(path)
	return generateSequence {
		map(reader.readLine())
	}
}

fun readPrimesFile(max: Int): Sequence<Int> {
	val fin = BufferedInputStream(FileInputStream("res/euler/primes9.bin"), 2 shl 20)
	val dis = DataInputStream(fin)
	return generateSequence {
		try {
			val p = dis.readInt()
			if (p > max) {
				dis.close()
				fin.close()
				return@generateSequence null
			}
			p
		} catch (e: EOFException) {
			dis.close()
			fin.close()
			return@generateSequence null
		}
	}
}
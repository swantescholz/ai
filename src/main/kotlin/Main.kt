
import ai.Ffnn
import extensions.sequences.ass
import extensions.toal
import math.linearalgebra.Vectord
import org.apache.commons.math3.fraction.BigFraction
import string.alof
import string.printl
import util.extensions.double
import java.math.BigInteger
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.util.*


fun mymain() {
	println("starting main...")
	printl(BigFraction(BigInteger.valueOf(100), BigInteger.valueOf(3)))
	val inList = alof(0, 0, 1, 0, 1, 1, 0, 1)
	val targets = alof(0, 0, 1, 0).map { it == 1 }
	val trainingVectors = ArrayList<Vectord>()
	inList.iterator().let { iterator ->
		while (iterator.hasNext())
			trainingVectors.add(Vectord(2, { iterator.next().double }))
	}
	printl(trainingVectors)
	printl(targets)
	val ffnn = Ffnn(2, 1, 3, 2)
	ffnn.initializeWeights()
	printl(ffnn.applyToTestInput(trainingVectors[0]))
	
}

fun main(args: Array<String>) {
	foo()
//	mymain()
}

fun foo() {
	fun enhanceParagraph(line: String): String {
		var res = line.replace("<em>", "*")
		res = res.replace("</em>", "*")
		res = res.replace("<strong>", "**")
		res = res.replace("</strong>", "**")
		res = res.replace("&#8211;", "-")
		res = res.replace("&#8230;", "...")
		res = res.replace("&#8217;", "'")
		res = res.replace("&#8216;", "'")
		res = res.replace("&#8216;", "'")
		res = res.replace("&#8220;", "\"")
		res = res.replace("&#8221;", "\"")
		res = res.replace("”", "\"")
		res = res.replace("“", "\"")
		res = res.replace(Regex("<.*?>"), "").trim()
		return res
	}
	
	fun getText(html: String): String {
		val lines = LinkedList(html.split("\n").ass().map { it.trim() }.toal())
		var title = ""
		while (true) {
			val line = lines.pop()
			if (line.startsWith("<h1 class=\"entry-title\">")) {
				title = line.substring("<h1 class=\"entry-title\">".length, line.length - 5)
				title = title.replace(Regex("<.*?>"), "").trim()
			}
			if (">Next Chapter<" in line || ">End<" in line || ">Last Chapter<" in line) {
				break
			}
		}
		val paragraphs = ArrayList<String>()
		while (true) {
			val line = lines.pop()
			if ("Next Chapter" in line || "Last Chapter" in line) {
				break
			}
			val paragraph = enhanceParagraph(line)
			paragraphs.add(paragraph)
		}
		var result = "# $title\n\n"
		result += paragraphs.joinToString("\n\n") + "\n\n=================================\n"
		return result
	}
	Files.write(Paths.get("worm.txt"), "".toByteArray(), StandardOpenOption.TRUNCATE_EXISTING)
	for (url in urls) {
		printl(url)
//		quitAfterTimes(3)
		//"https://parahumans.wordpress.com/2013/11/19/interlude-end/"
		val html = Scanner(URL(url).openStream(), "UTF-8").useDelimiter("\\A").next()
		val text = getText(html)
		Files.write(Paths.get("worm.txt"), text.toByteArray(), StandardOpenOption.APPEND)
	}
}

val urls = alof("https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-1-gestation/1-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-1-gestation/1-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-1-gestation/1-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-1-gestation/1-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-1-gestation/1-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-1-gestation/1-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-06/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-07/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-08/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-09/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-2-insinuation/2-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-06/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-07/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-08/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-09/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-10/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-11/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-12/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-3-agitation/3-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-06/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-07/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-08/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-09/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-10/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-11/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-x-bonus-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-4-shell/4-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-06/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-07/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-08/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-09/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-10/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-5-hive/5-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-06/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-07/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-08/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-09/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-6-tangle/6-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-01/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-02/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-03/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-04/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-05/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-06/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-07/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-08/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-09/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-10/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-11/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-12/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-7-buzz/7-x-interlude-arc-7-buzz/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-1/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-2/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-3/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-4/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-5/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-6/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-7/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-8/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-8-extermination/8-x-interlude/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-9-sentinel-interludes/9-1/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-9-sentinel-interludes/9-2/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-9-sentinel-interludes/9-3/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-9-sentinel-interludes/9-4/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-9-sentinel-interludes/9-5/","https://parahumans.wordpress.com/category/stories-arcs-1-10/arc-9-sentinel-interludes/9-6/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-1/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-2/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-3/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-4/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-5/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-6/","https://parahumans.wordpress.com/category/stories-arcs-1-10/%c2%ad-arc-10-parasite/10-x-bonus-interlude/Dragon","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-01/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-02/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-03/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-04/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-05/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-06/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-07/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-08-arc-11-infestation","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-a/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-b/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-c/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-d-arc-11-infestation","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-e/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-f/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-g/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-11-infestation-stories-arcs-11/11-h/","https://parahumans.wordpress.com/category/stories-arcs-11/arc-12-plague/12-01/","https://parahumans.wordpress.com/2012/06/30/plague-12-2/","https://parahumans.wordpress.com/2012/07/03/plague-12-3/","https://parahumans.wordpress.com/2012/07/07/plague-12-4/","https://parahumans.wordpress.com/2012/07/10/plague-12-5/","https://parahumans.wordpress.com/2012/07/14/plague-12-6/","https://parahumans.wordpress.com/2012/07/17/plague-12-7/","https://parahumans.wordpress.com/2012/07/21/plague-12-8/","https://parahumans.wordpress.com/2012/07/24/interlude-12/","https://parahumans.wordpress.com/2012/07/26/interlude-12½/","https://parahumans.wordpress.com/2012/07/28/snare-13-1/","https://parahumans.wordpress.com/2012/07/31/snare-13-2/","https://parahumans.wordpress.com/2012/08/02/interlude-13½-donation-bonus/","https://parahumans.wordpress.com/2012/08/04/snare-13-3/","https://parahumans.wordpress.com/2012/08/07/snare-13-4/","https://parahumans.wordpress.com/2012/08/11/snare-13-5/","https://parahumans.wordpress.com/2012/08/14/snare-13-6/","https://parahumans.wordpress.com/2012/08/18/snare-13-7/","https://parahumans.wordpress.com/2012/08/21/snare-13-8/","https://parahumans.wordpress.com/2012/08/25/snare-13-09/","https://parahumans.wordpress.com/2012/08/28/snare-13-10/","https://parahumans.wordpress.com/2012/09/01/interlude-13/","https://parahumans.wordpress.com/2012/09/04/prey-14-1/","https://parahumans.wordpress.com/2012/09/08/prey-14-2/","https://parahumans.wordpress.com/2012/09/11/prey-14-3/","https://parahumans.wordpress.com/2012/09/15/prey-14-4/","https://parahumans.wordpress.com/2012/09/18/prey-14-5/","https://parahumans.wordpress.com/2012/09/22/prey-14-6/","https://parahumans.wordpress.com/2012/09/25/prey-14-7/","https://parahumans.wordpress.com/2012/09/29/prey-14-8/","https://parahumans.wordpress.com/2012/10/02/prey-14-9/","https://parahumans.wordpress.com/2012/10/06/prey-14-10/","https://parahumans.wordpress.com/2012/10/09/prey-14-11/","https://parahumans.wordpress.com/2012/10/11/interlude-14/","https://parahumans.wordpress.com/2012/10/13/interlude-14-5-bonus-interlude/","https://parahumans.wordpress.com/2012/10/16/colony-15-1/","https://parahumans.wordpress.com/2012/10/18/interlude-15-donation-bonus/","https://parahumans.wordpress.com/2012/10/20/colony-15-2/","https://parahumans.wordpress.com/2012/10/23/colony-15-3/","https://parahumans.wordpress.com/2012/10/25/interlude-15-donation-bonus-2/","https://parahumans.wordpress.com/2012/10/27/colony-15-4/","https://parahumans.wordpress.com/2012/10/30/colony-15-5/","https://parahumans.wordpress.com/2012/11/03/colony-15-6/","https://parahumans.wordpress.com/2012/11/06/colony-15-7/","https://parahumans.wordpress.com/2012/11/08/interlude-15-donation-bonus-3/","https://parahumans.wordpress.com/2012/11/10/colony-15-8/","https://parahumans.wordpress.com/2012/11/13/colony-15-9/","https://parahumans.wordpress.com/2012/11/17/colony-15-10/","https://parahumans.wordpress.com/2012/11/20/interlude-15/","https://parahumans.wordpress.com/2012/11/24/monarch-16-1/","https://parahumans.wordpress.com/2012/11/27/monarch-16-2/","https://parahumans.wordpress.com/2012/11/29/interlude-16-donation-bonus/","https://parahumans.wordpress.com/2012/12/01/monarch-16-3/","https://parahumans.wordpress.com/2012/12/04/monarch-16-4/","https://parahumans.wordpress.com/2012/12/08/monarch-16-5/","https://parahumans.wordpress.com/2012/12/11/monarch-16-6/","https://parahumans.wordpress.com/2012/12/13/interlude-16-donation-bonus-2/","https://parahumans.wordpress.com/2012/12/15/monarch-16-7/","https://parahumans.wordpress.com/2012/12/18/monarch-16-8/","https://parahumans.wordpress.com/2012/12/22/monarch-16-9/","https://parahumans.wordpress.com/2012/12/25/monarch-16-10/","https://parahumans.wordpress.com/2012/12/27/interlude-16-donation-bonus-3/","https://parahumans.wordpress.com/2012/12/29/monarch-16-11/","https://parahumans.wordpress.com/2013/01/01/monarch-16-12/","https://parahumans.wordpress.com/2013/01/05/monarch-16-13/","https://parahumans.wordpress.com/2013/01/08/migration-17-1/","https://parahumans.wordpress.com/2013/01/09/migration-17-2/","https://parahumans.wordpress.com/2013/01/10/migration-17-3/","https://parahumans.wordpress.com/2013/01/11/migration-17-4/","https://parahumans.wordpress.com/2013/01/12/migration-17-5/","https://parahumans.wordpress.com/2013/01/13/migration-17-6/","https://parahumans.wordpress.com/2013/01/14/migration-17-7/","https://parahumans.wordpress.com/2013/01/15/migration-17-8/","https://parahumans.wordpress.com/2013/01/19/queen-18-1/","https://parahumans.wordpress.com/2013/01/22/queen-18-2/","https://parahumans.wordpress.com/2013/01/24/interlude-18x/","https://parahumans.wordpress.com/2013/01/26/queen-18-3/","https://parahumans.wordpress.com/2013/01/29/queen-18-4/","https://parahumans.wordpress.com/2013/01/31/interlude-18-donation-bonus-2","https://parahumans.wordpress.com/2013/02/02/queen-18-5/","https://parahumans.wordpress.com/2013/02/05/monarch-18-6/","https://parahumans.wordpress.com/2013/02/07/interlude-18-donation-bonus-3/","https://parahumans.wordpress.com/2013/02/09/queen-18-7/","https://parahumans.wordpress.com/2013/02/12/queen-18-8/","https://parahumans.wordpress.com/2013/02/14/interlude-18-donation-bonus-4","https://parahumans.wordpress.com/2013/02/16/interlude-18/","https://parahumans.wordpress.com/2013/02/19/scourge-19-1/","https://parahumans.wordpress.com/2013/02/23/scourge-19-2/","https://parahumans.wordpress.com/2013/02/26/scourge-19-3/","https://parahumans.wordpress.com/2013/02/28/interlude-19-donation-bonus-1/","https://parahumans.wordpress.com/2013/03/02/scourge-19-4/","https://parahumans.wordpress.com/2013/03/05/scourge-19-5/","https://parahumans.wordpress.com/2013/03/09/scourge-19-6/","https://parahumans.wordpress.com/2013/03/12/scourge-19-7/","https://parahumans.wordpress.com/2013/03/16/interlude-19-y/","https://parahumans.wordpress.com/2013/03/19/interlude-19/","https://parahumans.wordpress.com/2013/03/21/chrysalis-20-1/","https://parahumans.wordpress.com/2013/03/23/chrysalis-20-2/","https://parahumans.wordpress.com/2013/03/26/chrysalis-20-3/","https://parahumans.wordpress.com/2013/03/30/chrysalis-20-4/","https://parahumans.wordpress.com/2013/04/02/chrysalis-20-5/","https://parahumans.wordpress.com/2013/04/04/interlude-20-donation-bonus-1/","https://parahumans.wordpress.com/2013/04/06/interlude-20/","https://parahumans.wordpress.com/2013/04/09/imago-21-1/","https://parahumans.wordpress.com/2013/04/13/imago-21-2/","https://parahumans.wordpress.com/2013/04/16/imago-21-3/","https://parahumans.wordpress.com/2013/04/18/imago-21-4/","https://parahumans.wordpress.com/2013/04/20/imago-21-5/","https://parahumans.wordpress.com/2013/04/23/imago-21-6/","https://parahumans.wordpress.com/2013/04/25/imago-21-7/","https://parahumans.wordpress.com/2013/04/27/interlude-21-donation-bonus-1/","https://parahumans.wordpress.com/2013/04/30/interlude-21/","https://parahumans.wordpress.com/2013/05/04/cell-22-1/","https://parahumans.wordpress.com/2013/05/07/cell-22-2/","https://parahumans.wordpress.com/2013/05/09/cell-22-3/","https://parahumans.wordpress.com/2013/05/11/cell-22-4","https://parahumans.wordpress.com/2013/05/14/cell-22-5/","https://parahumans.wordpress.com/2013/05/16/cell-22-6/","https://parahumans.wordpress.com/2013/05/18/interlude-22/","https://parahumans.wordpress.com/2013/05/21/interlude-22-donation-bonus-1/","https://parahumans.wordpress.com/2013/05/25/drone-23-1/","https://parahumans.wordpress.com/2013/05/28/drone-23-2/","https://parahumans.wordpress.com/2013/05/30/drone-23-3/","https://parahumans.wordpress.com/2013/06/01/drone-23-4/","https://parahumans.wordpress.com/2013/06/04/drone-23-5/","https://parahumans.wordpress.com/2013/06/06/interlude-23/","https://parahumans.wordpress.com/2013/06/08/crushed-24-1/","https://parahumans.wordpress.com/2013/06/11/crushed-24-2/","https://parahumans.wordpress.com/2013/06/15/crushed-24-3/","https://parahumans.wordpress.com/2013/06/18/crushed-24-4/","https://parahumans.wordpress.com/2013/06/20/crushed-24-5/","https://parahumans.wordpress.com/2013/06/22/interlude-24/","https://parahumans.wordpress.com/2013/06/25/interlude-24-donation-bonus-1/","https://parahumans.wordpress.com/2013/06/29/scarab-25-1/","https://parahumans.wordpress.com/2013/07/02/scarab-25-2/","https://parahumans.wordpress.com/2013/07/06/scarab-25-3/","https://parahumans.wordpress.com/2013/07/09/scarab-25-4/","https://parahumans.wordpress.com/2013/07/11/scarab-25-5/","https://parahumans.wordpress.com/2013/07/13/scarab-25-6/","https://parahumans.wordpress.com/2013/07/16/interlude-25/","https://parahumans.wordpress.com/2013/07/18/sting-26-1/","https://parahumans.wordpress.com/2013/07/20/sting-26-2/","https://parahumans.wordpress.com/2013/07/23/sting-26-3/","https://parahumans.wordpress.com/2013/07/25/interlude-26-donation-bonus-1/","https://parahumans.wordpress.com/2013/07/27/sting-26-4/","https://parahumans.wordpress.com/2013/07/30/sting-26-5/","https://parahumans.wordpress.com/2013/08/03/sting-26-6/","https://parahumans.wordpress.com/2013/08/06/interlude-26a/","https://parahumans.wordpress.com/2013/08/08/interlude-26b/","https://parahumans.wordpress.com/2013/08/10/interlude-26/","https://parahumans.wordpress.com/2013/08/13/extinction-27-1/","https://parahumans.wordpress.com/2013/08/17/extinction-27-2/","https://parahumans.wordpress.com/2013/08/20/extinction-27-3/","https://parahumans.wordpress.com/2013/08/22/extinction-27-4/","https://parahumans.wordpress.com/2013/08/24/extinction-27-5/","https://parahumans.wordpress.com/2013/08/27/interlude-27/","https://parahumans.wordpress.com/2013/08/29/interlude-27b/","https://parahumans.wordpress.com/2013/08/31/cockroaches-28-1/","https://parahumans.wordpress.com/2013/09/03/cockroaches-28-2/","https://parahumans.wordpress.com/2013/09/05/cockroaches-28-3/","https://parahumans.wordpress.com/2013/09/07/cockroaches-28-4/","https://parahumans.wordpress.com/2013/09/10/cockroaches-28-5/","https://parahumans.wordpress.com/2013/09/14/cockroaches-28-6/","https://parahumans.wordpress.com/2013/09/17/interlude-28/","https://parahumans.wordpress.com/2013/09/19/venom-29-1/","https://parahumans.wordpress.com/2013/09/21/venom-29-2/","https://parahumans.wordpress.com/2013/09/24/venom-29-3/","https://parahumans.wordpress.com/2013/09/26/venom-29-4/","https://parahumans.wordpress.com/2013/09/28/venom-29-5/","https://parahumans.wordpress.com/2013/10/01/venom-29-6/","https://parahumans.wordpress.com/2013/10/03/venom-29-7/","https://parahumans.wordpress.com/2013/10/05/venom-29-8/","https://parahumans.wordpress.com/2013/10/08/venom-29-9/","https://parahumans.wordpress.com/2013/10/12/interlude-29/","https://parahumans.wordpress.com/2013/10/15/speck-30-1/","https://parahumans.wordpress.com/2013/10/17/speck-30-2/","https://parahumans.wordpress.com/2013/10/19/speck-30-3/","https://parahumans.wordpress.com/2013/10/22/speck-30-4/","https://parahumans.wordpress.com/2013/10/24/speck-30-5/","https://parahumans.wordpress.com/2013/10/26/speck-30-6/","https://parahumans.wordpress.com/2013/10/29/30-7/","https://parahumans.wordpress.com/2013/11/02/teneral-e-1/","https://parahumans.wordpress.com/2013/11/05/teneral-e-2/","https://parahumans.wordpress.com/2013/11/09/teneral-e-3/","https://parahumans.wordpress.com/2013/11/12/teneral-e-4/","https://parahumans.wordpress.com/2013/11/16/teneral-e-5/","https://parahumans.wordpress.com/2013/11/19/interlude-end/")
import javafx.application.Application
import javafx.event.ActionEvent
import javafx.event.EventHandler
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.layout.StackPane
import javafx.stage.Stage

class Gui : Application() {
	
	override fun start(primaryStage: Stage) {
		primaryStage.title = "Hello World!"
		val root = StackPane()
		val btn = Button()
		btn.text = "Say 'Hello World'"
		btn.onAction = EventHandler<ActionEvent> { println("Hello World!")
			val tmp = Button()
			tmp.text = "Say 'Hello World 222'"
			root.children.add(tmp)
		}
		
		
		root.children.add(btn)
		primaryStage.scene = Scene(root, 300.0, 250.0)
		primaryStage.show()
	}
	
	
	fun myLaunch() {
		Application.launch()
	}
	
}

fun main(args: Array<String>) {
	Gui().myLaunch()
}
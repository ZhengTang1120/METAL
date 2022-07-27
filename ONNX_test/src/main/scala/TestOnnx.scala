import java.io.{FileWriter, PrintWriter}

import com.typesafe.config.ConfigFactory

import scala.io.Source

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.slf4j.{Logger, LoggerFactory}

import java.time.LocalDateTime
import java.time.Duration

import scala.io.Source


object TestOnnx extends App {

  val start_time = LocalDateTime.now()

  val ortEnvironment = OrtEnvironment.getEnvironment
  val modelpath1 = "/data1/home/zheng/METAL/best_model.onnx"
  val session1 = ortEnvironment.createSession(modelpath1, new OrtSession.SessionOptions)

  println(session1.getOutputInfo)

  for (words<-??){
    val word_input = new java.util.HashMap[String, OnnxTensor]()
    word_input.put("words",  OnnxTensor.createTensor(ortEnvironment, words.toArray))
    val emissionScores = session1.run(word_input).get(0).getValue.asInstanceOf[Array[Array[Float]]]
  }

  println(Duration.between(start_time, LocalDateTime.now()).getSeconds)

}
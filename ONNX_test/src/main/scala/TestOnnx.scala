import java.io.{FileWriter, PrintWriter}

import com.typesafe.config.ConfigFactory

import scala.io.Source

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import org.slf4j.{Logger, LoggerFactory}

import java.time.LocalDateTime
import java.time.Duration

import scala.io.Source

import scala.util.parsing.json._


object TestOnnx extends App {



  val ortEnvironment = OrtEnvironment.getEnvironment
  val modelpath1 = "/data1/home/zheng/METAL/best_model.onnx"
  val session1 = ortEnvironment.createSession(modelpath1, new OrtSession.SessionOptions)

  println(session1.getOutputInfo)

  val start_time = LocalDateTime.now()

  val jsonString = Source.fromFile("/data1/home/zheng/METAL/word_ids.json").getLines.mkString
  val parsed = JSON.parseFull(jsonString)

  for (line<-parsed.get.asInstanceOf[List[Any]]){
    val words = line.asInstanceOf[List[Any]].map(i => i.asInstanceOf[Number].longValue)
    val word_input = new java.util.HashMap[String, OnnxTensor]()
    word_input.put("words",  OnnxTensor.createTensor(ortEnvironment, words.toArray))
    val emissionScores = session1.run(word_input).get(0).getValue.asInstanceOf[Array[Array[Float]]]
  }

  println(Duration.between(start_time, LocalDateTime.now()).getSeconds)

}
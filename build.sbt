name := "lda-spark"

version := "0.1"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "org.scalanlp" % "chalk" % "1.3.0",
  "org.apache.spark" %% "spark-core" % "1.0.0-SNAPSHOT",
  "org.apache.spark" %% "spark-mllib" % "1.0.0-SNAPSHOT",
  "org.apache.hadoop" % "hadoop-client" % "1.0.4",
  "org.scalatest"    %% "scalatest" % "1.9.1"  % "test"
)

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

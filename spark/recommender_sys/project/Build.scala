import sbt.Keys._
import sbt._

object MyBuild extends Build {
  val sigOptSparkLibGit = ProjectRef(uri("https://github.com/sigopt/sigopt-spark.git"), "sigopt-spark")
  lazy val root = Project("root", file("."))
     .dependsOn(sigOptSparkLibGit)
     .settings(
        libraryDependencies <++= (scalaVersion) { scalaVersion =>
            val v = scalaVersion match {
                case twoTen if scalaVersion.startsWith("2.10") => "_2.10"
                case twoEleven if scalaVersion.startsWith("2.11") => "_2.11"
                case _ => "_" + scalaVersion
        }
        Seq(
             "com.sigopt"                  %  ("sigopt-java") % "3.1.1",
             "org.apache.spark"            %  ("spark-core" + v) % "1.6.1",
             "org.apache.spark"            %  ("spark-mllib" + v) % "1.6.1",
             "org.json4s"                  %  ("json4s-jackson" + v) % "3.2.10"
        )
    })
}

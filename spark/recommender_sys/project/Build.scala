import sbt.Keys._
import sbt._

object MyBuild extends Build {
  lazy val root = Project("root", file("."))
     .settings(
        libraryDependencies <++= (scalaVersion) { scalaVersion =>
            Seq(
                 "org.apache.spark"            %%  ("spark-core") % "1.6.1" % "provided",
                 "org.apache.spark"            %%  ("spark-mllib") % "1.6.1" % "provided",
                 "com.sigopt"                  %  ("sigopt-java") % "3.4.0"
            )
        }
      )
}

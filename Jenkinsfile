@Library('pio-lib@v9') _
pipeline {

    agent {
        label 'linux'
    }

    options {
        timeout(time: 4, unit: 'HOURS')
    }

    environment {
        ARTIFACTORY_PUSH = credentials('JenkinsArtifactoryUser')
        // Isolate the gcloud configuration used in the test stage from other builds running
        // on the same Jenkins slave. Technically this isn't an issue right now because we
        // only have one executor per slave but that could change at any moment.
        CLOUDSDK_CONFIG="${WORKSPACE}"
    }

    stages {
        stage('Test'){
            steps {
                withCredentials([
                    file(
                        credentialsId: 'jenkins-at-google-cloud-storage',
                        variable: 'GOOGLE_APPLICATION_CREDENTIALS'
                    )  // This is to access buckets/objects in Google Cloud Storage (project id: prowlerio-pf-sandbox).
                ]) {
                    poetrySetup()
                    // The venv activation should not be necessary and it is in fact coupled to
                    // the implementation details of poetrySetup(). This is unfortunate,
                    // but necessary for now.
                    // See https://prowlerio.atlassian.net/browse/PTKB-9216 for more details.
                    sh '''
                        . .venv/bin/activate
                        poetry install
                        mkdir pio_home
                        export PROWLER_IO_HOME=pio_home
                        poetry run task test
                       '''
                }
            }
        }
        stage('Archive Test Output'){
            steps {
                publishHTML([
                    allowMissing: true,
                    alwaysLinkToLastBuild: false,
                    keepAll: true,
                    reportDir: 'cover_html',
                    reportFiles: 'index.html',
                    reportName:  'Coverage Report',
                    reportTitles: 'Coverage Report'
                ])
            }
        }
    }

    post {
        always {
            junit '**/reports/*.xml'
            cleanWs notFailBuild: true
        }
    }
}

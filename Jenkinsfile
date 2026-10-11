// Objective 6 – CI/CD with Jenkins: test -> train -> gate -> build image -> deploy to Kubernetes
pipeline {
  agent any
  options { timestamps(); timeout(time: 60, unit: 'MINUTES') }
  environment {
    IMAGE    = "voyage-api"
    REGISTRY = credentials('docker-registry-url')   // e.g. docker.io/<user>  (optional)
    PYTHONPATH = "${WORKSPACE}/src"
  }
  stages {
    stage('Setup') {
      steps {
        sh 'python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt'
      }
    }
    stage('Validate data + unit tests') {
      steps {
        sh '. .venv/bin/activate && python -m voyage.validate data && pytest -q tests'
      }
    }
    stage('Train + track (MLflow)') {
      steps { sh '. .venv/bin/activate && python -m voyage.train_all' }
    }
    stage('Quality gate') {
      steps { sh '. .venv/bin/activate && python -m voyage.validate gate' }
    }
    stage('Build Docker image') {
      steps {
        sh 'docker build -t $IMAGE:$GIT_COMMIT -t $IMAGE:latest .'
        sh 'docker build -f Dockerfile.streamlit -t voyage-dashboard:$GIT_COMMIT -t voyage-dashboard:latest .'
      }
    }
    stage('Smoke test container') {
      steps {
        sh '''
          docker rm -f voyage-smoke || true
          docker run -d --name voyage-smoke -p 5055:5000 $IMAGE:$GIT_COMMIT
          for i in $(seq 1 20); do curl -fs localhost:5055/ready && break || sleep 2; done
          curl -fs -X POST localhost:5055/predict/price -H 'Content-Type: application/json' \
            -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"economic","agency":"CloudFy"}'
        '''
      }
      post { always { sh 'docker rm -f voyage-smoke || true' } }
    }
    stage('Deploy to Kubernetes') {
      when { branch 'main' }
      steps {
        sh '''
          kubectl apply -f k8s/
          kubectl -n voyage set image deployment/voyage-api api=$IMAGE:$GIT_COMMIT
          kubectl -n voyage rollout status deployment/voyage-api --timeout=180s
        '''
      }
    }
  }
  post {
    always { archiveArtifacts artifacts: 'models/metrics.json', allowEmptyArchive: true }
    failure { echo 'Pipeline failed – see the stage logs above.' }
  }
}

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#데이터셋 로드
iris = load_iris()
x = iris.data[:,2:] # 꽃잎의 길이, 너비
y = iris.target

x_train,x_test,y_train,y_test = train_test_split( x,y,test_size=0.3,random_state=2021,shuffle=True)



#약한 학습기 구축
log_model = LogisticRegression()
rnd_model = RandomForestClassifier()
svm_model = SVC()

#앙상블 모델 구축
voting_model = VotingClassifier (
    estimators = [('lr',log_model),
                ('rf',rnd_model),
                ('svc',svm_model)],
                voting='hard' # 직접 투표 방법
)

#앙상블 모델 학습
voting_model.fit (x_train,y_train)

#모델 비교
for model in (log_model,rnd_model,svm_model,voting_model):
  model.fit(x_train,y_train)
  y_pred =model.predict(x_test)
  print(model.__class__.__name__,":",accuracy_score(y_test,y_pred))



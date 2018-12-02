
#함수	설명
tf.shape	텐서의 구조를 알아냅니다.
tf.size	텐서의 크기를 알아냅니다.
tf.rank	텐서의 랭크를 알아냅니다.
tf.reshape	텐서의 엘리먼트(element)는 그대로 유지하면서 텐서의 구조를 바꿉니다.
tf.squeeze	텐서에서 크기가 1인 차원을 삭제합니다.
tf.expand_dims	텐서에 차원을 추가합니다.
tf.slice	텐서의 일부분을 삭제합니다.
tf.split	텐서를 한 차원을 기준으로 여러개의 텐서로 나눕니다.
tf.tile	한 텐서를 여러번 중복으로 늘려 새 텐서를 만듭니다.
tf.concat	한 차원을 기준으로 텐서를 이어 붙입니다.
tf.reverse	텐서의 지정된 차원을 역전시킵니다.
tf.transpose	텐서를 전치(transpose)시킵니다. 행열바꾸기
tf.gather	주어진 인덱스에 따라 텐서의 엘리먼트를 모읍니다.

#상수
tf.zeros_like	모든 엘리먼트를 0으로 초기화한 텐서를 생성합니다.
tf.ones_like	모든 엘리먼트를 1로 초기화한 텐서를 생성합니다.
tf.fill	주어진 스칼라(scalar) 값으로 엘리먼트를 초기화한 텐서를 생성합니다.
tf.constant	함수 인자로 지정된 값을 이용하여 상수 텐서를 생성합니다.

#파라미터

함수	설명
tf.random_normal	정규 분포 형태의 갖는 난수 텐서를 생성합니다.
tf.truncated_normal	정규 분포 형태의 난수 텐서를 생성합니다. 다만 2표준편차 범위 밖의 값을 제거합니다.
tf.random_uniform	균등 분포 형태의 난수 텐서를 생성합니다.
tf.random_shuffle	첫번째 차원을 기준으로 하여 텐서의 엘리먼트를 섞습니다.
tf.set_random_seed	난수 시드(seed)를 제공합니다.
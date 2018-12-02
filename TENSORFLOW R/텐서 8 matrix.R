library(tensorflow)
library(magrittr)

sess<-tf$InteractiveSession()

#creating matrix
matrix1<-tf$constant(matrix(c(1:6),ncol = 2)) #int32
matrix2<-tf$constant(matrix(c(1:6),ncol = 2),dtype = tf$float32) #float32
matrix3<-tf$constant(matrix(as.numeric(c(1:6)),ncol = 2)) #float64

#실제값 확인
matrix1$eval()
matrix2$eval()
matrix3$eval()

matrix1 <- aperm(array(c(1:24), dim = c(4, 3, 2)), perm = c(2, 1, 3)) %>% 
  tf$constant()
matrix1$eval()

# 3D array 기준: 
tf$reduce_sum(matrix1, axis = 0L)$eval()  # 세로방향
tf$reduce_sum(matrix1, axis = 1L)$eval()  # 가로방향
tf$reduce_sum(matrix1, axis = 2L)$eval()  # 레이어방향

# -의 의미는 정의된 방향 중 '뒤에서부터'를 의미함.
# 즉, -1L은 뒤에서부터 첫번째 방향인 레이어방향
tf$reduce_sum(matrix1, axis = -1L)$eval() # 레이어방향
tf$reduce_sum(matrix1, axis = -2L)$eval() # 가로방향
tf$reduce_sum(matrix1, axis = -3L)$eval() # 세로방향


tf$reduce_sum(matrix1)$eval()

tf$shape(matrix1) # 행,열,레이어 순서

# tensor reshape: -1L의 의미는 "알아서" 하라는 의미.
# 다음 명령어의 의미는 2D 행렬로 변환하되,
# 열수는 4개, 행갯수는 알아서 바꿔라라는 의미.
matrix2 <- tf$reshape(matrix1, shape = c(-1L, 4L))
matrix2$eval() # 우리의 예상대로 나오지 않음.

#R함수 이용하기
matrix3 <- rbind(matrix1[,,0]$eval(), matrix1[,,1]$eval()) %>%
  tf$constant(dtype = "float32")
matrix3$eval()

#tensorflow 함수 이용하기 
matrix4 <- tf$concat(c(matrix1[,,0], matrix1[,,1]), axis = 0L)
matrix4$eval()

# squeeze와 expand_dim 함수
matrix4 <- tf$expand_dims(tf$transpose(matrix4), 0L)
matrix4$eval()
matrix4 <- tf$squeeze(tf$transpose(matrix4))
matrix4$eval()

# onehot coding
onehot <- tf$one_hot(as.integer(c(0,1,2,0)), depth = 3L)
onehot$eval()

# Close session
sess$close()

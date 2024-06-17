package ch04_.statement.sec04.forloop;
/*
 * 부동소숫점의 float 타입의 연산 과정에서 0.1을 정확하게 표현하지 못한다 
 */
public class FloatCounterExample {
	public static void main(String[] args) {
		for(float x=0.1f; x<=1.0f; x+=0.1f) { // 9번만 반복
			System.out.println(x);
		}
	}
}

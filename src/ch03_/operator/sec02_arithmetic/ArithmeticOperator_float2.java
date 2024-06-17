/*
 * 산술연산자 : +, -, *, /, %
 * 나머지 연산자(%)를 사용하지 않고 몫과 나머지를 구하라 
 * 위 
 */

package ch03_.operator.sec02_arithmetic;
// static 변수 + static 메서드 
// 객체 생성 필요 없이 클래스명으로 바로 사용 
// e.g. book.price = 2000 (o) /  Book b1 = new Book() 객체 생성 (x)
public class ArithmeticOperator_float2 {
	public static void main(String[] args) {
		float x = 10.0f;
		float y = 3.0f;
		
		float m = (float)(int)(x / y); // 3.333333 -> 정수 변환 3 (casting 강제타입 변환  
		float r = x - (m * y);
		System.out.printf("몫(%f), 나머지(%f)\n", m, r);
	}
}	





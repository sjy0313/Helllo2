/*
 * 산술연산자 : +, -, *, /, %
 */

package ch03_.operator.sec02_arithmetic;

public class ArithmeticOperatorExample_float {
	public static void main(String[] args) {
		float f1 = 10.0f; // 10.0f
		float f2 = 3.3f;  // 2.5f
		
		float m3 = f1 / f2;
		float f3 = f1 % f2; // % : 나머지 
		System.out.printf("몫 : (%f) = (%f) %% (%f)\n", m3, f1, f2);
		System.out.printf("나머지 : (%f) = (%f) %% (%f)\n", f3, f1, f2);
		// 나머지 : (0.100000) = (10.000000) % (3.300000)
		
		float f4 = f1 / f2; // f4(3.030303) // float 타입의 정밀도 한계로 인해 근사값 저장 
		float f5 = f4 * f2; // f5(10.000000)  
		double f6 = Math.IEEEremainder(f1, f2);
		
		System.out.printf("IEEEremainder : f6(%f) = f1(%f) %% f2(%f)\n", f6, f1, f2);
		System.out.printf("f4(%f), f5(%f)\n", f4, f5); // printf 함수는 기본적으로 소수점 아래
		// 6자리까지 출력하며 필요에 따라 반올림함 . 
		
	}
}	


package ch02_.variable_type.sec04_Float;

public class FloatDoubleExample {
	public static void main(String[] args) {
		/*정밀도 확인 IEEE 754 표준에 근거해서 float 타입과 double 타입의
		 값을 부동 소수점 방식으로 메모리에 저장*/
		// float : 4byte f or F
		// double : 8byte 기본형
		
		float var1 = 0.1234567890123456789f; // 유효 소수점 이하자리 : 7자리
		double var2 = 0.1234567890123456789; // 유효 소수점 이하자리 : 15자리가 안전 
		System.out.println("var1: " + var1);
		System.out.println("var2: " + var2);
		// 지수형태 : e or E 
		//10의 거듭제곱 리터럴
		double var3 = 3e6; // 3 * 10의 6승, 3,000,000,0
		float var4 = 3e6F;
		double var5 = 2e-3; // 2.0의 10의 -3승, 0.002 
		System.out.println("var3: " + var3);
		System.out.println("var4: " + var4);
		System.out.println("var5: " + var5);
	}
 }

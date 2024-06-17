package ch02_.variable_type.sec02_integer;
/*
 * 정수타입 : int
 * 4byte(32bit)
 * 정수타입의 기본타입(byte/short/char/int/long)
 * CPU 구조상 32bit가 모든 연산에 가장 효율적임.
 * 효율을 높이기 위해 자료형 지정필요*/
public class IntegerLiteralExample {
	public static void main(String[] args) {
		// 부호를 가짐.
		int var1 = 0b1011; //2진수 0b/0B 로 시작 0과 1로 자성 
		int var2 = 0206; //8진수  
		int var3 = 365; //10진수
		int var4 = 0xB3; //16진수 

		System.out.println("var1: " + var1);
		System.out.println("var2: " + var2);
		System.out.println("var3: " + var3);
		System.out.println("var4: " + var4);
	}
}
package ch02_.variable_type.sec02_integer;

public class LongExample {
	public static void main(String[] args) {
		long var1 = 10; //10
		long var2 = 20L; //20
		//long var3 = 1000000000000; //컴파일러는 int로 간주하기 때문에 에러 발생
		//The literal 1000000000000 of type int is out of range //
		long var4 = 1234567890123L; // 숫자 뒤에 ㅣ, L을 붙여 long타입 값을 컴파일러에게 알려주어야함.
		long iMax = 2147483647; // 정수(int)의 가장 큰 값을 long에 할당 

		System.out.println(var1);
		System.out.println(var2);
		System.out.println(var4);
	}
}
package ch02_.variable_type.sec02_integer;

public class ByteExample2 {
	public static void main(String[] args) {
		//강제 자료형 변환(캐스팅)
		byte var1 = (byte)128;  
		byte var2 = (byte)0b10000000; /*2 진수 -> 10진수 변환 최상위 비트가 1인 음수인
		* 경우 나머지 7개의 bit를 모두 1의 보수(즉, 1은 0, 0은 1로 변환한 후 더한 값에
		*  -를 붙히면 10진수가 됨)*/
		
		
		byte var3 = (byte)0x80;
		System.out.println(var1); // -128
		System.out.println(var2); // -128
		System.out.println(var3); // -128
	}
}
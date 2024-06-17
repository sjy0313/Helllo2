package ch02_.variable_type.sec08_Casting;

// 강제 타입 변환(casting)
// 자료형이 큰 타입에서 작은 타입으로 강제 변환 
// 변수 = (자료형)변수
public class CastingExample {
	public static void main(String[] args) {
		// int -> byte 
		int var1 = 10;
		byte var2 = (byte) var1;
		System.out.println(var2);	 //강제 타입 변환 후에 10이 그대로 유지
		// long -> int 
		long var3 = 300;
		int var4 = (int) var3;
		System.out.println(var4);	 //강제 타입 변환 후에 300이 그대로 유지
		// int -> char 
		int var5 = 65;
		char var6 = (char) var5;
		System.out.println(var6); 	//'A'가 출력
		// double -> int
		//소숫점 이하 짤림 
		double var7 = 3.14;
		int var8 = (int) var7;
		System.out.println(var8); 	//3이 출력
	}
}

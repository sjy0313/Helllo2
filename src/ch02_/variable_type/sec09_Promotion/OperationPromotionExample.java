package ch02_.variable_type.sec09_Promotion;

public class OperationPromotionExample {
	public static void main(String[] args) {
		byte result1 = 10 + 20; //컴파일 단계에서 연산
		System.out.println("result1: " + result1);

		byte v1 = 10;
		byte v2 = 20;
		int result2 = v1 + v2; //int 타입으로 변환 후 연산
		System.out.println("result2: " + result2);

		byte v3 = 10;
		int v4 = 100;
		long v5 = 1000L ;
		// int 타입보다 허용 범위(255까지)가 더 큰 long 타입이 피연산자로 사용되면
		// 다른 피연산자는 long 타입으로 변환되어 연산을 수행한다 
		// v5가 long 타입이므로 모두 long 타입으로 변환 후 연산 
		//int result3 = v3 + v4 + v5; ->Type mismatch: cannot convert from long to int
		long result3 = v3 + v4 + v5; //long 타입으로 변환 후 연산
		System.out.println("result3: " + result3);
			
		char v6 = 'A';
		char v7 = 1;
		int result4 = v6 + v7; //int 타입으로 변환 후 연산
		System.out.println("result4: " + result4);
		System.out.println("result4: " + (char)result4);

		int v8 = 10;
		int result5 = v8 / 4; //정수 연산의 결과는 정수
		System.out.println("result5: " + result5);

		int v9 = 10;
		double result6 = v9 / 4.0; //double 타입으로 변환 후 연산
		System.out.println("result6: " + result6);

		int v10 = 1;
		int v11 = 2;
		double result7 = (double) v10 / v11; // 1/2(유리수)double 타입으로 변환 후 연산
		System.out.println("result7: " + result7); //0.5
		
		double result8 = v10 / (double)v11; // double 타입으로 변환 후 연산 수행 
		System.out.println("result8: " + result8); //0.5
		
		double result9
		= (double)(v10 / v11); // int 로 연산을 수행 후 double 타입
		System.out.println("resul9: " + result9); // 0.0 
		
		
	}
	
}
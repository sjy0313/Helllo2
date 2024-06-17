package ch02_.variable_type.sec07_Autotype_converter;

public class PromotionExample {
	public static void main(String[] args) {
		//자동 타입 변환 : promotion 
		//크기가 더 작은 자료형을 더 큰 자료형에 대입할 때, 자동으로 작은 자료형이
		//큰 자료형으로 변환되는 현상
	


		// 값의 허용 범위가 작은 타입이 허용 범위가 큰 타입으로 대입될 떄 발생
		byte byteValue = 10;
		int intValue = byteValue;
		System.out.println("intValue: " + intValue);
		
		// char -> int 
		char charValue = '가';
		intValue = charValue;
		System.out.println("가의 유니코드: " + intValue);
		// int -> long 
		intValue = 50;
		long longValue = intValue;
		System.out.println("longValue: " + longValue);
		
		// long(8byte) -> float(4byte)
		longValue = 100;
		float floatValue = longValue;
		System.out.println("floatValue: " + floatValue);
		// float -> double
		floatValue = 100.5F;
		double doubleValue = floatValue;
		System.out.println("doubleValue: " + doubleValue);
	}
}			
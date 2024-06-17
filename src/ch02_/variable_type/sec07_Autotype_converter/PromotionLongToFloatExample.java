package ch02_.variable_type.sec07_Autotype_converter;

public class PromotionLongToFloatExample {
	public static void main(String[] args) {
		//자동 타입 변환 : promotion
		// 값의 허용 범위가 작은 타입이 허용 범위가 큰 타입으로 대입될 떄 발생
		// long(8byte) -> float(4byte)
		//long 변수는 8바이트, float 변수는 4바이트의 공간을 할당받지만,
		//float 변수가 더 크다고 되어있다.
		/* 이는 기본적으로 표현할 수 있는 숫자의 범위가 더 넓은 실수형 변수가 정수형
		변수보다 크다고 정의하기 때문이다*/

		long longValue = 123456789123L; // floatValue: 1.23456791E11
		float floatValue = longValue; // 오차발생 
		System.out.println("floatValue: " + floatValue); // 1234567890 -> floatValue: 1.23456794E9
	}
}

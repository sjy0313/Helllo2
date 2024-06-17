package ch02_.variable_type.sec10_String_Conversion;
// 문자열을 기본 타입으로 변환 
// Rapper class : Boolean, Byte, Short, Character, Integer, Long, Float, Double
public class PrimitiveAndStringConversionExample {
	public static void main(String[] args) {
		
		
		// 강제 캐스팅 형태로 자료형을 지원하지 않음
		// String str = "10";
		// int nval = (int)str;
		int value1 = Integer.parseInt("10"); // 매소드(static(정적인 함수) 
		// 일반적인 인스턴스 변수들과는 달리 프로그램 로딩시에 static영역의
		// 메모리에 올라가기 때문에 객체 생성없이 사용할 수 있습니다. 
	
		double value2 = Double.parseDouble("3.14");
		boolean value3 = Boolean.parseBoolean("true");
		
		System.out.println("value1: " + value1);
		System.out.println("value2: " + value2);
		System.out.println("value3: " + value3);
		
		String str1 = String.valueOf(10);	// 정수 -> 문자열
		String str2 = String.valueOf(3.14); // 실수 -> 문자열
		String str3 = String.valueOf(true);	// 불리언 -> 문자열 
		
		System.out.println("str1: " + str1);
		System.out.println("str2: " + str2);
		System.out.println("str3: " + str3);
	}
}
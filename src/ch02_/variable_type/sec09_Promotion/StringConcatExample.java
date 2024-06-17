package ch02_.variable_type.sec09_Promotion;
// 문자열 결합 + 연산자 
// 피연산 중 하나가 문자열일 경우에는 나머지 피연산자도 
// 
public class StringConcatExample {
	public static void main(String[] args) {
		//숫자 연산
		int result1 = 10 + 2 + 8;
		System.out.println("result1: " + result1);

		
		//결합 연산
		// 10과 2가 먼저 산술연산 후 문자열로 변환되어 "8"과 결합
		// 10 + 2 -> 12 -> "12"+"8" -> "128"
		String result2 = 10 + 2 + "8"; 
		System.out.println("result2: " + result2);

		String result3 = 10 + "2" + 8;
		System.out.println("result3: " + result3);
		// 문자열이 먼저 나와서 2가 ""변환되어 "102"
		// "102"와 8이 "8"로 변환 후 "1028"로 결합
		String result4 = "10" + 2 + 8; // 1028
		System.out.println("result4: " + result4);

		String result5 = "10" + (2 + 8); // 괄호안의 숫자가 산술 연산 수행 후 문저열 결합  
		System.out.println("result5: " + result5); //1010
	}
}
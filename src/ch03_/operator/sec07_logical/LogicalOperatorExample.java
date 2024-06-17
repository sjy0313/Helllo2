
/*
 * 논리연산자 : 논리곱(&&, &), 논리합(||, |), 배타적 논리합(^), 논리부정(!)
 */

package ch03_.operator.sec07_logical;

public class LogicalOperatorExample {
	public static void main(String[] args) {
		int charCode = 'A';
		int charZ = 'Z';
		int char_a = 'a';
		int char_z = 'z';
		
		System.out.printf("('%c) -> (%d)\n", charCode, charCode);
		System.out.printf("('%c) -> (%d)\n", charZ, charZ);
		System.out.printf("('%c) -> (%d)\n", char_a, char_a);
		System.out.printf("('%c) -> (%d)\n", char_z, char_z);
		
		
		//int charCode = 'a';
		//int charCode = '5';

		if( (65<=charCode) & (charCode<=90) ) {
			System.out.println("대문자이군요.");
			
		}
			
		if( (97<=charCode) && (charCode<=122) ) {
			System.out.println("소문자이군요.");
		}

		if( (48<=charCode) && (charCode<=57) ) {
			System.out.println("0~9 숫자이군요.");
		}
		
		//----------------------------------------------------------

		int value = 6;
		//int value = 7;
			
		if( (value%2==0) | (value%3==0) ) {  // | = or 
			System.out.println("2 또는 3의 배수이군요.");
		}

		boolean result = (value%2==0) || (value%3==0);
		if( !result ) { // result가 거짓(false)이면 실행 
			System.out.println("2 또는 3의 배수가 아니군요.");
		}
	}
}
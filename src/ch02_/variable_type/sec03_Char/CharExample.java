package ch02_.variable_type.sec03_Char;

public class CharExample {
	public static void main(String[] args) {
		// char : 2바이트, 0~65535
		char c1 = 'A';          	//문자 저장
		char c2 = 65;          	//유니코드 직접 저장
		
		char c3 = '가';         	//문자 저장(java는 보통 2byte로 문자처리)
		char c4 = 44032;      	//유니코드 직접 저장
		
		// 알파벳 같은 경우 1byte로 충분하지만 다국어가 사용되며 2byte로 처리하게 되면서
		// 용량이 늘었지만 그만큼 연산처리의 일관성을 높이면서 효율적이게됨.
		char c5 = "가";
		// Type mismatch: cannot convert from String to char
		
		System.out.println(c1);
		System.out.println(c2);
		System.out.println(c3);
		System.out.println(c4);
	}
} 

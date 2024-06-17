package ch05_.references.sec05.strings;

public class CharAtExample {
	public static void main(String[] args) {
		String ssn = "9506241230123";
		// 6번쨰 요소 -> 1 
		// 문자열의 요소참조를 배열형태로 할 수 없다 
		// System.out.println(ssn[0]);
		
		char sex = ssn.charAt(6);
		switch (sex) {
			case '1':
			case '3':
				System.out.println("남자입니다.");
				break;
			case '2':
			case '4':
				System.out.println("여자입니다.");
				break;
		}
	}
}
package ch05_.references.sec05.strings;

public class SubStringExample {
	public static void main(String[] args) {
		String ssn = "880815-1234567";
			
		String firstNum = ssn.substring(0, 6);
		System.out.print(firstNum); // prtinln (은 출력 후 자동으로 줄바꿈)
		
		String middleChar = ssn.substring(6,7);
		System.out.print(middleChar);
		/*public String substring(int beginIndex, int endIndex) {*/
		
		// begin index 부터, length() 전까지
		String secondNum = ssn.substring(7); // 시작 위치만 제공
		/*public String substring(int beginIndex) {
	        return substring(beginIndex, length());*/
		System.out.print(secondNum);
	}
}
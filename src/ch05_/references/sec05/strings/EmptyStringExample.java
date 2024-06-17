package ch05_.references.sec05.strings;

public class EmptyStringExample {
	public static void main(String[] args) {
		String hobby = ""; // 빈무자열 : 객체가 할당 됨, null을 할당한 것과 다름 
		
		if(hobby.equals("")) {
			System.out.println("hobby 변수가 참조하는 String 객체는 빈 문자열");
		}
	}
}
package ch07_.inheritance.sec03_parent_constructor.exam02;

public class Phone {
	//필드 선언
	public String model;
	public String color;
	// 아무런 생성자를 만들지 않으면 compiler 가 자동으로 기본생성자 생성해줌.
	/*
	 * public Phone() {
	 * 
}
	 */
	//매개변수를 갖는 생성자 선언
	
	
	public Phone(String model, String color) {
		this.model = model;
		this.color = color;
		System.out.println("Phone(String model, String color) 생성자 실행");
	}
}
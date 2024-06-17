package ch07_.inheritance.sec03_parent_constructor.exam01;
// 최상위 클래스는 object
public class SmartPhone extends Phone {
	//자식 생성자 선언
	public SmartPhone(String model, String color) {
	//  모든 객체는 생성자를 호출해야하는데 
		super(); // python에서 __init__ 부모(디폴트)객체의 생성자 호출 
		// 부모기본 생성자 호출(phone.java)
		this.model = model;
		this.color = color;
		System.out.println("SmartPhone(String model, String color) 생성자 실행됨");
	}
}
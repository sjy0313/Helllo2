package ch07_.inheritance.sec03_parent_constructor.exam03;

public class SmartPhone extends Phone {
	
	/* 자식의 기본 생성자를 정의하면
	 * - 부모도 기본 생성자를 정의
	 * - 자식이 부모의 다른 생성자를 호출(revoke)
	 */
	public SmartPhone() {
		//super();
		super(null, null);
	}
	
	//자식 생성자 선언
	public SmartPhone(String model, String color) {
		super(model, color);
		System.out.println("SmartPhone(String model, String color) 생성자 실행됨");
	}

	
}

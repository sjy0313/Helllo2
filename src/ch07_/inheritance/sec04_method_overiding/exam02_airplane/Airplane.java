package ch07_.inheritance.sec04_method_overiding.exam02_airplane;
/*
  오버로딩:

같은 클래스 내에서 사용.
매개변수의 타입, 개수, 순서를 다르게 하여 동일한 이름의 
메서드를 여러 개 정의.
컴파일 타임에 결정.

오버라이딩:

상속 관계에 있는 클래스에서 사용.
상위 클래스의 메서드를 하위 클래스에서 재정의.
런타임에 결정.
 */
public class Airplane {
	//메소드 선언
	public void land() {
		System.out.println("착륙합니다.");
}

	public void fly() {
		System.out.println("일반 비행합니다.");
	}

	public void takeOff() {
		System.out.println("이륙합니다.");
	}
}
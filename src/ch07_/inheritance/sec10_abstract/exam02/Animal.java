package ch07_.inheritance.sec10_abstract.exam02;
/*
 * abstract method : abstract 리턴타입 메소드이름(파라미터);
 * 추상클래스는 실체 클래스의 부모역할을 하므로 실제 클래스는 
 * 추상클래스를 상속해서 공통적인 필드나 메소드를 물려받을 수 있음.
 */
// 추상클래스 선언[new 연산자를 이용하여 객체를 직접 만들지 못하고 상속을 통해 자식 클래스만 만들 수 있음]
public abstract class Animal {
	//메소드 선언
	public void breathe() {
		System.out.println("숨을 쉽니다.");
	}

	//추상 메소드 선언
	public abstract void sound();
	// body 가 있으면 추상클래스 조건위반되어 애러
}
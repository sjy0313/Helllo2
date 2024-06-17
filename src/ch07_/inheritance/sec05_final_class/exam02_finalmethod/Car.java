package ch07_.inheritance.sec05_final_class.exam02_finalmethod;
/* final method + class 는 
 * final method :
 * - 자식 클래스(subclass)에서 오버라이딩(overriding)할 수 없다
 * 오버라이딩(overriding) 즉 재정의 할 수 없음.
 * 
 * stop method 를 final method로 선언했기 떄문에 sportcar에서 stop() 오버라이딩불가
 */
public class Car {
	//필드 선언
	public int speed;

	//메소드 선언
	public void speedUp() {
		speed += 1;
	}

	//final 메소드
	public final void stop() {
		System.out.println("차를 멈춤");
		speed = 0;
	}
}
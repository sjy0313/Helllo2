package ch06_.classes.sec14.getter_setter;
/*
 * Getter & Setter
 * - 이름규칙 : get이나 set에 속성의 첫글자는 대문자로 지정 후 나머지 긁자 지정
 * - 자료형이 boolean이면 get 대신 is 
 * - 예 : 속성(speed) / getter(getSpeed) / setter(setSpeed)
 * - 이 규칙은 자바 프레임워크에서 대부분 요구하는 사항
 * - javabeans(java framework), WAS(Web Application Server)의 자바빈 등록 규칙
 */
public class Car {
	//필드 선언 
	private int speed;
	private boolean stop;
	
	//speed 필드의 Getter/Setter 선언
	// 가진 속성값을 리턴(getter)
	public int getSpeed() {
		return speed;
	}
	public void setSpeed(int speed) {
		if(speed < 0) {
			this.speed = 0;
			return;
		} else {
			this.speed = speed;
		}
	}
	//stop 필드의 Getter/Setter 선언
	public boolean isStop() {
		return stop;
	}
	public void setStop(boolean stop) {
		this.stop = stop;
		if(this.stop == true) {
			this.speed = 0;
		}
	}
}